# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import math

import torch

from .attention import Attention, FeedForward
from .rope import LTXRopeType, precompute_freqs_cis
from .rope import _generate_freq_grid_np as generate_freq_grid_np
from .rope import _generate_freq_grid_pytorch as generate_freq_grid_pytorch
from .utils_ltx2 import rms_norm


class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = rms_norm(hidden_states)
        norm_hidden_states = norm_hidden_states.squeeze(1)
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        norm_hidden_states = rms_norm(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    """1D transformer-based connector for sequential embeddings."""

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        )
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=apply_gated_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0
                - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = hidden_states.shape
        assert S % self.num_learnable_registers == 0

        num_registers_duplications = S // self.num_learnable_registers
        learnable_registers = torch.tile(
            self.learnable_registers, (num_registers_duplications, 1)
        ).to(hidden_states.dtype)  # [S, D]

        # [B, S] binary: True for valid tokens, False for padding
        mask_2d = attention_mask.squeeze(1).squeeze(1) >= -9000.0

        results = []
        for b in range(B):
            valid_mask = mask_2d[b]  # [S]
            valid_tokens = hidden_states[b, valid_mask, :]  # [N_valid, D]
            pad_length = S - valid_tokens.shape[0]
            padded = torch.nn.functional.pad(
                valid_tokens, pad=(0, 0, 0, pad_length), value=0
            )  # [S, D]
            flipped = torch.flip(
                valid_mask.to(hidden_states.dtype).unsqueeze(-1), dims=[0]
            )  # [S, 1]
            results.append(flipped * padded + (1 - flipped) * learnable_registers)

        hidden_states = torch.stack(results, dim=0)
        attention_mask = torch.full_like(attention_mask, 0.0)
        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states, attention_mask
            )

        indices_grid = torch.arange(
            hidden_states.shape[1],
            dtype=torch.float32,
            device=hidden_states.device,
        )
        indices_grid = indices_grid[None, None, :]
        freq_grid_generator = (
            generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        )
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)
        return hidden_states, attention_mask


class Embeddings1DConnectorConfigurator:
    @classmethod
    def _from_transformer_config(
        cls, config: dict, *, audio: bool = False
    ) -> Embeddings1DConnector:
        config = config.get("transformer", {})
        rope_type = LTXRopeType(config.get("rope_type", "interleaved"))
        double_precision_rope = config.get("frequencies_precision", False) == "float64"
        pe_max_pos = config.get("connector_positional_embedding_max_pos", [1])

        if audio:
            num_attention_heads = config.get(
                "audio_connector_num_attention_heads",
                config.get("connector_num_attention_heads", 30),
            )
            attention_head_dim = config.get(
                "audio_connector_attention_head_dim",
                config.get("connector_attention_head_dim", 128),
            )
            num_layers = config.get(
                "audio_connector_num_layers",
                config.get("connector_num_layers", 2),
            )
        else:
            num_attention_heads = config.get("connector_num_attention_heads", 30)
            attention_head_dim = config.get("connector_attention_head_dim", 128)
            num_layers = config.get("connector_num_layers", 2)

        return Embeddings1DConnector(
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            num_learnable_registers=config.get("connector_num_learnable_registers", 128),
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=config.get("connector_apply_gated_attention", False),
        )

    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        return cls._from_transformer_config(config)


class AudioEmbeddings1DConnectorConfigurator(Embeddings1DConnectorConfigurator):
    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        return cls._from_transformer_config(config, audio=True)


def _norm_and_concat_per_token_rms(encoded_text: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-token RMS normalization used by LTX-2.3 text projection."""
    if encoded_text.ndim != 4:
        raise ValueError(
            "LTX-2.3 text projection expects stacked hidden states with shape "
            f"[batch, seq, hidden, layers], got {tuple(encoded_text.shape)}"
        )
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    batch_size, seq_len, hidden_dim, num_layers = encoded_text.shape
    normed = normed.reshape(batch_size, seq_len, hidden_dim * num_layers)
    mask = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)


class GemmaFeaturesExtractorProjLinear(torch.nn.Module):
    """Gemma text projection for LTX-2 connector inputs."""

    _GEMMA3_12B_HIDDEN_SIZE = 3840
    _GEMMA3_12B_NUM_LAYERS_WITH_EMBED = 49

    def __init__(
        self,
        *,
        video_inner_dim: int | None = None,
        audio_inner_dim: int | None = None,
        use_v2_projection: bool = False,
    ) -> None:
        super().__init__()
        self.use_v2_projection = use_v2_projection
        self.embedding_dim = self._GEMMA3_12B_HIDDEN_SIZE
        flat_dim = self.embedding_dim * self._GEMMA3_12B_NUM_LAYERS_WITH_EMBED
        if use_v2_projection:
            self.video_aggregate_embed = torch.nn.Linear(
                flat_dim,
                video_inner_dim or self.embedding_dim,
                bias=True,
            )
            self.audio_aggregate_embed = torch.nn.Linear(
                flat_dim,
                audio_inner_dim or self.embedding_dim,
                bias=True,
            )
        else:
            self.aggregate_embed = torch.nn.Linear(flat_dim, self.embedding_dim, bias=False)

    @property
    def uses_raw_hidden_states(self) -> bool:
        return self.use_v2_projection

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if not self.use_v2_projection:
            return self.aggregate_embed(x)
        if attention_mask is None:
            raise ValueError("LTX-2.3 text projection requires an attention mask")
        normed = _norm_and_concat_per_token_rms(x, attention_mask).to(x.dtype)
        video_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(_rescale_norm(normed, video_dim, self.embedding_dim))
        audio = None
        if self.audio_aggregate_embed is not None:
            audio_dim = self.audio_aggregate_embed.out_features
            audio = self.audio_aggregate_embed(_rescale_norm(normed, audio_dim, self.embedding_dim))
        return video, audio

    @classmethod
    def from_config(cls, config: dict) -> "GemmaFeaturesExtractorProjLinear":
        transformer_config = config.get("transformer", {})
        if transformer_config.get("caption_proj_before_connector", False):
            video_inner_dim = transformer_config.get("num_attention_heads", 32) * transformer_config.get(
                "attention_head_dim", 128
            )
            audio_inner_dim = transformer_config.get(
                "audio_num_attention_heads",
                transformer_config.get("num_attention_heads", 32),
            ) * transformer_config.get(
                "audio_attention_head_dim",
                transformer_config.get("attention_head_dim", 128),
            )
            return cls(
                video_inner_dim=video_inner_dim,
                audio_inner_dim=audio_inner_dim,
                use_v2_projection=True,
            )
        return cls()
