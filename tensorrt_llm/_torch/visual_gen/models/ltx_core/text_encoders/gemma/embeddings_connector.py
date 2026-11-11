import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.model_protocol import ModelConfigurator
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.attention import Attention
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.feed_forward import FeedForward
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.utils import rms_norm


class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        apply_gated_attention: bool = False,
        ff_bias: bool = True,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )

        self.ff = FeedForward(
            dim,
            dim_out=dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention — `mask` is the kernel-boundary name for the additive mask.
        attn_output = self.attn1(norm_hidden_states, mask=additive_attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    """
    Embeddings1DConnector applies a 1D transformer-based processing to sequential embeddings (e.g., for video, audio, or
    other modalities). It supports rotary positional encoding (rope), optional causal temporal positioning, and can
    substitute padded positions with learnable registers. The module is highly configurable for head size, number of
    layers, and register usage.
    Args:
        attention_head_dim (int): Dimension of each attention head (default=128).
        num_attention_heads (int): Number of attention heads (default=30).
        num_layers (int): Number of transformer layers (default=2).
        positional_embedding_theta (float): Scaling factor for position embedding (default=10000.0).
        positional_embedding_max_pos (list[int] | None): Max positions for positional embeddings (default=[1]).
        causal_temporal_positioning (bool): If True, uses causal attention (default=False).
        num_learnable_registers (int | None): Number of learnable registers to replace padded tokens. If None, disables
            register replacement. (default=128)
        rope_type (LTXRopeType): The RoPE variant to use.
        double_precision_rope (bool): Use double precision rope calculation (default=False).
    """

    _supports_gradient_checkpointing = True

    def __init__(  # noqa: PLR0913
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        ff_bias: bool = True,
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
                    ff_bias=ff_bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, additive_attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        assert seq_len % self.num_learnable_registers == 0

        registers = self.learnable_registers.repeat(seq_len // self.num_learnable_registers, 1).to(hidden_states.dtype)
        registers = registers.unsqueeze(0).expand(batch_size, -1, -1)  # (B, seq_len, hidden_dim)
        binary_mask = additive_attention_mask[:, 0, 0, :].unsqueeze(-1) >= 0
        binary_mask = binary_mask.to(hidden_states.dtype)
        hidden_states = binary_mask * hidden_states + (1 - binary_mask) * registers

        return hidden_states, torch.zeros_like(additive_attention_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Embeddings1DConnector.
        Args:
            hidden_states: (B, S, D) input embeddings.
            additive_attention_mask: optional additive mask of shape (B, 1, 1, S), where
                valid = 0.0 and padding = -torch.finfo(dtype).max.
        Returns:
            (hidden_states, additive_attention_mask)
        """
        if self.num_learnable_registers:
            hidden_states, additive_attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states, additive_attention_mask
            )

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :].expand(hidden_states.shape[0], -1, -1)
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
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
            hidden_states = block(hidden_states, additive_attention_mask=additive_attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)

        return hidden_states, additive_attention_mask


class Embeddings1DConnectorConfigurator(ModelConfigurator[Embeddings1DConnector]):
    """Configurator for video embeddings connector."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> Embeddings1DConnector:
        transformer_config = metadata.get("config", {}).get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "split"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

        # Video connector dimensions
        num_attention_heads = transformer_config.get("connector_num_attention_heads", 30)
        attention_head_dim = transformer_config.get("connector_attention_head_dim", 128)
        num_layers = transformer_config.get("connector_num_layers", 2)

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
            ff_bias=transformer_config.get("connector_ff_bias", True),
        )
        return connector


class AudioEmbeddings1DConnectorConfigurator(ModelConfigurator[Embeddings1DConnector]):
    """Configurator for audio embeddings connector with separate dimension settings."""

    @classmethod
    def from_metadata(cls, metadata: dict) -> Embeddings1DConnector:
        transformer_config = metadata.get("config", {}).get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "split"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

        # Audio connector dimensions - fall back to video connector config for backwards compatibility
        num_attention_heads = transformer_config.get(
            "audio_connector_num_attention_heads",
            transformer_config.get("connector_num_attention_heads", 30),
        )
        attention_head_dim = transformer_config.get(
            "audio_connector_attention_head_dim",
            transformer_config.get("connector_attention_head_dim", 128),
        )
        num_layers = transformer_config.get(
            "audio_connector_num_layers",
            transformer_config.get("connector_num_layers", 2),
        )

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
            ff_bias=transformer_config.get("connector_ff_bias", True),
        )
        return connector
