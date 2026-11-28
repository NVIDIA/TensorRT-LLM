# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native image-only H3 transformer: shared H3 blocks and a readout per PDD step."""

import math

import torch
from torch import nn

from tensorrt_llm.quantization.mode import QuantAlgo

from ...config import DiffusionModelConfig
from ..minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3StaticContext,
    MiniMaxH3TokenRefiner,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
)
from ..modeling import BaseDiffusionModel
from .schedule import NUM_PDD_STEPS


class MiniMaxH3ImageTransformer2DModel(MiniMaxH3Transformer3DModel):
    """Image specialization of the native H3 checkpoint loader and text refiner.

    It has no audio inputs, audio parameters or video output head. All six
    image readouts are stored in the converted transformer checkpoint itself.
    """

    def __init__(self, model_config: DiffusionModelConfig):
        # The AV constructor allocates extra projections which are absent from
        # the image checkpoint. Only the common H3 blocks are constructed here.
        BaseDiffusionModel.__init__(self, model_config)
        cfg = self.config = model_config.pretrained_config
        if cfg.num_pdd_heads != NUM_PDD_STEPS or tuple(cfg.patch_size) != (1, 2, 2):
            raise ValueError("Expected the six-head MiniMax-H3 image checkpoint with 1x2x2 patches")
        if model_config.mapping.world_size != 1 or model_config.attention.sparse_attention_config:
            raise ValueError("MiniMax-H3-Image currently requires dense, single-device attention")
        quant_algo = model_config.quant_config.quant_algo
        if quant_algo is not None and not (
            quant_algo in (QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.NVFP4)
            and model_config.dynamic_weight_quant
        ):
            raise ValueError(
                "MiniMax-H3-Image supports mixed BF16/FP32 weights or dynamically quantized "
                "FP8 / FP8_BLOCK_SCALES / NVFP4 weights only"
            )
        if quant_algo == QuantAlgo.NVFP4 and not model_config.force_dynamic_quantization:
            raise ValueError(
                "MiniMax-H3-Image NVFP4 requires force_dynamic_quantization=True; "
                "the dynamic weight loader does not calibrate activation scales"
            )
        # Quantize the shared H3 blocks, keeping image projections and timestep
        # embedding in their checkpoint precision (ordinary torch.nn.Linear).
        self.proj_in = nn.Linear(cfg.in_channels * 4, cfg.hidden_size, dtype=torch.float32)
        self.context_embedder = nn.Linear(
            cfg.text_dim, cfg.hidden_size, dtype=model_config.torch_dtype
        )
        self.time_embedder = nn.Sequential()
        self.time_embedder.add_module(
            "linear_1", nn.Linear(cfg.freq_dim, cfg.time_embed_hidden_dim, dtype=torch.float32)
        )
        self.time_embedder.add_module("act", nn.SiLU())
        self.time_embedder.add_module(
            "linear_2",
            nn.Linear(cfg.time_embed_hidden_dim, cfg.time_embed_dim, dtype=torch.float32),
        )
        self.rope = MiniMaxH3RotaryPosEmbed(cfg.rope_freq_dim, cfg.rope_theta)
        shared = dict(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            attention_head_dim=cfg.attention_head_dim,
            ffn_dim=cfg.ffn_dim,
            norm_eps=cfg.norm_eps,
            qk_norm_eps=cfg.qk_norm_eps,
            model_config=model_config,
        )
        self.token_refiner = MiniMaxH3TokenRefiner(
            num_layers=cfg.num_refiner_layers, final_norm_eps=cfg.final_norm_eps, **shared
        )
        self.transformer_blocks = nn.ModuleList(
            MiniMaxH3TransformerBlock(layer_idx=i, time_embed_dim=cfg.time_embed_dim, **shared)
            for i in range(cfg.num_layers)
        )
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            cfg.hidden_size, cfg.time_embed_dim, cfg.final_norm_eps, model_config
        )
        self.pdd_proj_out = nn.ModuleList(
            nn.Linear(cfg.hidden_size, cfg.in_channels * 4, dtype=torch.float32)
            for _ in range(NUM_PDD_STEPS)
        )
        self.apply_quant_config_exclude_modules()

    def post_load_weights(self) -> None:
        from tensorrt_llm._torch.modules.linear import Linear

        for module in self.modules():
            if isinstance(module, Linear):
                module.post_load_weights()
        # Meta-init may not preserve non-persistent buffers during materialization.
        self.rope = MiniMaxH3RotaryPosEmbed(self.config.rope_freq_dim, self.config.rope_theta).to(
            self.device
        )

    def prepare_static_context(
        self, encoder_hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> MiniMaxH3StaticContext:
        text = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        return MiniMaxH3StaticContext(self.token_refiner(text), self.rope(position_ids))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        static_context: MiniMaxH3StaticContext,
        pdd_step_index: int,
    ) -> torch.Tensor:
        """Text-to-image: all rows have the same noise level; text rows precede image rows."""
        if not 0 <= pdd_step_index < len(self.pdd_proj_out) or timestep.numel() != 1:
            raise ValueError("Expected one noise level and an in-range PDD head index")
        text = static_context.text_embeds
        hidden = torch.cat((text, self.proj_in(hidden_states.float()).to(text.dtype)), dim=1)
        if hidden.shape[1] != static_context.sequence_length:
            raise ValueError("RoPE coordinates must cover exactly the packed text and image rows")
        tags = torch.zeros(hidden.shape[1], device=hidden.device, dtype=torch.long)
        tags[: text.shape[1]] = 1
        # H3 conditions on data-ward time, not on descending noise (timestep).
        half = self.config.freq_dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, device=hidden.device, dtype=torch.float32
        )
        angles = (1 - timestep.float()).reshape(1, 1) * torch.exp(exponent / half)
        temb = self.time_embedder(torch.cat((angles.cos(), angles.sin()), dim=-1))
        for block in self.transformer_blocks:
            hidden = block(hidden, temb, tags, static_context.rotary_emb, timestep=timestep)
        hidden = self.norm_out(hidden, temb, torch.zeros_like(tags))
        return self.pdd_proj_out[pdd_step_index](hidden.float())[:, text.shape[1] :]
