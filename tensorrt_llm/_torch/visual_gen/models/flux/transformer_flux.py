"""FLUX Transformer model implementation.

FLUX Architecture:
- 19 dual-stream blocks (FluxTransformerBlock): Separate processing for image/text
- 38 single-stream blocks (FluxSingleTransformerBlock): Joint processing
- Joint attention mechanism with QK normalization
- 2D RoPE position embeddings

Forward Pass Flow:
1. Embed inputs: x_embedder(latents), context_embedder(text), time_text_embed(timestep, pooled)
2. Compute RoPE from position IDs
3. Run 19 dual-stream blocks: (hidden_states, encoder_hidden_states) processed separately
4. Run 38 single-stream blocks: Concatenate, process, split
5. norm_out + proj_out -> noise prediction
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.models.flux.attention_flux import FluxAttention, FluxPosEmbed

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization for DiT blocks.

    Computes: norm(x) * (1 + scale) + shift, with gating for attention and MLP.

    Returns 5 modulation parameters:
    - norm_hidden_states: Normalized and shifted/scaled hidden states
    - gate_msa: Gate for multi-head self-attention output
    - shift_mlp, scale_mlp: Shift and scale for MLP input
    - gate_mlp: Gate for MLP output
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = False,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        # TRT-LLM Linear for quantization support
        self.linear = Linear(
            embedding_dim,
            6 * embedding_dim,
            bias=True,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        # TRT-LLM LayerNorm (elementwise_affine=False → has_weights=False, has_bias=False)
        self.norm = LayerNorm(
            hidden_size=embedding_dim,
            eps=eps,
            has_weights=elementwise_affine,
            has_bias=elementwise_affine,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq, dim)
            emb: Timestep embedding (batch, dim)

        Returns:
            Tuple of (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        """
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Simplified AdaLN for single-stream blocks.

    Returns only normalized hidden states and a gate.
    """

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        # TRT-LLM Linear for quantization support
        self.linear = Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=True,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        # TRT-LLM LayerNorm (no learnable params)
        self.norm = LayerNorm(
            hidden_size=embedding_dim, eps=eps, has_weights=False, has_bias=False, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq, dim)
            emb: Timestep embedding (batch, dim)

        Returns:
            Tuple of (norm_hidden_states, gate)
        """
        emb = self.linear(self.silu(emb))
        shift, scale, gate = emb.chunk(3, dim=1)

        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x, gate


class AdaLayerNormContinuous(nn.Module):
    """Continuous adaptive layer normalization for output projection."""

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = False,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        # TRT-LLM Linear for quantization support
        self.linear = Linear(
            conditioning_embedding_dim,
            2 * embedding_dim,
            bias=True,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        # TRT-LLM LayerNorm
        self.norm = LayerNorm(
            hidden_size=embedding_dim,
            eps=eps,
            has_weights=elementwise_affine,
            has_bias=elementwise_affine,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq, dim)
            emb: Conditioning embedding (batch, dim)

        Returns:
            Normalized and modulated tensor
        """
        emb = self.linear(self.silu(emb))
        scale, shift = emb.chunk(2, dim=1)  # HF order: scale first, then shift

        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combined timestep and text projection embeddings (for schnell).

    Uses diffusers embedding classes for compatibility. These small layers
    don't need quantization - the bulk of compute is in transformer blocks.
    Note: Dtype conversion happens via model.to(dtype) after weight loading.
    """

    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        **kwargs,  # Accept but ignore dtype/quant params for API compatibility
    ):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Combined timestep, guidance, and text projection embeddings (for dev).

    Uses diffusers embedding classes for compatibility. These small layers
    don't need quantization - the bulk of compute is in transformer blocks.
    Note: Dtype conversion happens via model.to(dtype) after weight loading.
    """

    def __init__(
        self,
        embedding_dim: int,
        pooled_projection_dim: int,
        **kwargs,  # Accept but ignore dtype/quant params for API compatibility
    ):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        pooled_projection: torch.Tensor,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))

        time_guidance_emb = timesteps_emb + guidance_emb
        pooled_projections = self.text_embedder(pooled_projection)

        return time_guidance_emb + pooled_projections


class GELU(nn.Module):
    """GELU activation with input projection.

    This matches HuggingFace diffusers' GELU class which wraps the projection
    inside the activation module. This ensures weight names match:
    - `ff.net.0.proj.weight` instead of `ff.net.0.weight`
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        # TRT-LLM Linear for quantization support
        self.proj = Linear(
            dim_in,
            dim_out,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class FeedForward(nn.Module):
    """GELU feed-forward network matching HuggingFace diffusers structure.

    Weight naming matches HuggingFace:
    - net.0: GELU module with proj (input projection)
    - net.1: Dropout
    - net.2: Linear (output projection)
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Match HuggingFace FeedForward structure
        approximate = "tanh" if activation_fn == "gelu-approximate" else "none"
        act_fn = GELU(
            dim,
            inner_dim,
            approximate=approximate,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )

        self.net = nn.ModuleList(
            [
                act_fn,  # net.0 (GELU with proj)
                nn.Dropout(dropout),  # net.1
                Linear(  # net.2 - TRT-LLM Linear
                    inner_dim,
                    dim_out,
                    bias=bias,
                    dtype=dtype,
                    quant_config=quant_config,
                    skip_create_weights_in_init=skip_create_weights,
                    force_dynamic_quantization=force_dynamic_quant,
                    disable_deep_gemm=True,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class FluxTransformerBlock(nn.Module):
    """Dual-stream transformer block for FLUX.

    Processes image and text tokens separately, combines via joint attention,
    then applies separate FFNs.

    Architecture:
    1. AdaLN for image (norm1) and text (norm1_context)
    2. Joint attention (FluxAttention with added_kv_proj_dim)
    3. Residual + gated attention output
    4. LayerNorm + modulation for FFN
    5. Separate FFNs for image (ff) and text (ff_context)
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # AdaLN for image and text
        self.norm1 = AdaLayerNormZero(
            dim,
            eps=eps,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )
        self.norm1_context = AdaLayerNormZero(
            dim,
            eps=eps,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )

        # Joint attention
        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,  # Enables joint attention
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
            config=config,
            layer_idx=layer_idx,
        )

        # FFN normalization (TRT-LLM LayerNorm)
        self.norm2 = LayerNorm(
            hidden_size=dim, eps=1e-6, has_weights=False, has_bias=False, dtype=dtype
        )
        self.norm2_context = LayerNorm(
            hidden_size=dim, eps=1e-6, has_weights=False, has_bias=False, dtype=dtype
        )

        # FFN layers
        self.ff = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )
        self.ff_context = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: Image tokens (batch, img_seq, dim)
            encoder_hidden_states: Text tokens (batch, txt_seq, dim)
            temb: Timestep embedding (batch, dim)
            image_rotary_emb: RoPE (cos, sin) tuple
            joint_attention_kwargs: Additional kwargs for attention

        Returns:
            Tuple of (encoder_hidden_states, hidden_states)
        """
        # Image: AdaLN modulation
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        # Text: AdaLN modulation
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # Joint attention
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Image: Gated residual for attention
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # Image: FFN with modulation
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Text: Gated residual for attention
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # Text: FFN with modulation
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        # FP16 overflow protection
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream transformer block for FLUX.

    Concatenates image and text tokens, processes together,
    then splits back.

    Architecture:
    1. Concatenate encoder_hidden_states + hidden_states
    2. AdaLayerNormZeroSingle
    3. Parallel attention + MLP branches
    4. proj_out(concat(attn, mlp))
    5. Gated residual
    6. Split back into image and text
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # AdaLN
        self.norm = AdaLayerNormZeroSingle(
            dim,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )

        # MLP branch (TRT-LLM Linear for quantization support)
        self.proj_mlp = Linear(
            dim,
            self.mlp_hidden_dim,
            bias=True,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        self.act_mlp = nn.GELU(approximate="tanh")

        # Output projection (concat of attn + mlp) - TRT-LLM Linear
        self.proj_out = Linear(
            dim + self.mlp_hidden_dim,
            dim,
            bias=True,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )

        # Attention (no added_kv_proj_dim since tokens are already concatenated)
        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,  # No output projection in attention
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
            config=config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: Image tokens (batch, img_seq, dim)
            encoder_hidden_states: Text tokens (batch, txt_seq, dim)
            temb: Timestep embedding (batch, dim)
            image_rotary_emb: RoPE (cos, sin) tuple
            joint_attention_kwargs: Additional kwargs for attention

        Returns:
            Tuple of (encoder_hidden_states, hidden_states)
        """
        text_seq_len = encoder_hidden_states.shape[1]

        # Concatenate text + image
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states

        # AdaLN
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        # MLP branch
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        # Attention branch
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Concat and project
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)

        # Residual
        hidden_states = residual + hidden_states

        # FP16 overflow protection
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        # Split back into text and image
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(nn.Module):
    """FLUX Transformer model for text-to-image generation.

    This is the native TRT-LLM implementation of FLUX transformer.
    Supports FP8/NVFP4 quantization for optimized inference.

    Architecture:
    - pos_embed: FluxPosEmbed for 2D RoPE
    - time_text_embed: Combined timestep + guidance + text projection embeddings
    - context_embedder: Linear projection for T5 text embeddings
    - x_embedder: Linear projection for latent inputs
    - transformer_blocks: 19 dual-stream FluxTransformerBlock
    - single_transformer_blocks: 38 single-stream FluxSingleTransformerBlock
    - norm_out: AdaLayerNormContinuous
    - proj_out: Linear projection to output
    """

    def __init__(self, model_config: "DiffusionModelConfig"):
        super().__init__()
        self.model_config = model_config

        # Extract pretrained config from model_config
        pretrained_config = model_config.pretrained_config

        # Extract dtype and quantization config for Linear modules (following Wan pattern)
        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        # Extract FLUX-specific parameters from pretrained config
        num_attention_heads = getattr(pretrained_config, "num_attention_heads", 24)
        attention_head_dim = getattr(pretrained_config, "attention_head_dim", 128)
        in_channels = getattr(pretrained_config, "in_channels", 64)
        out_channels = getattr(pretrained_config, "out_channels", in_channels)
        num_layers = getattr(pretrained_config, "num_layers", 19)
        num_single_layers = getattr(pretrained_config, "num_single_layers", 38)
        joint_attention_dim = getattr(pretrained_config, "joint_attention_dim", 4096)
        pooled_projection_dim = getattr(pretrained_config, "pooled_projection_dim", 768)
        guidance_embeds = getattr(pretrained_config, "guidance_embeds", False)
        patch_size = getattr(pretrained_config, "patch_size", 1)
        axes_dims_rope = getattr(pretrained_config, "axes_dims_rope", [16, 56, 56])
        theta_rope = getattr(pretrained_config, "theta", 10000)

        # Compute inner dimension
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds

        # Store config for compatibility
        self.config = type(
            "Config",
            (),
            {
                "num_attention_heads": num_attention_heads,
                "attention_head_dim": attention_head_dim,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_layers": num_layers,
                "num_single_layers": num_single_layers,
                "joint_attention_dim": joint_attention_dim,
                "pooled_projection_dim": pooled_projection_dim,
                "guidance_embeds": guidance_embeds,
                "patch_size": patch_size,
                "axes_dims_rope": axes_dims_rope,
                "theta_rope": theta_rope,
                "inner_dim": self.inner_dim,
            },
        )()

        # Position embeddings
        self.pos_embed = FluxPosEmbed(theta=theta_rope, axes_dim=list(axes_dims_rope))

        # Time + text embeddings
        if guidance_embeds:
            self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=pooled_projection_dim,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights=skip_create_weights,
                force_dynamic_quant=force_dynamic_quant,
            )
        else:
            self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim,
                pooled_projection_dim=pooled_projection_dim,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights=skip_create_weights,
                force_dynamic_quant=force_dynamic_quant,
            )

        # Input embedders (TRT-LLM Linear for quantization support)
        self.context_embedder = Linear(
            joint_attention_dim,
            self.inner_dim,
            bias=True,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        # NOTE: x_embedder quantization is disabled when in_channels < 128.
        # FLUX.1 has in_channels=64, which is below the 128-block size required by
        # fp8_block_scaling_gemm (causes NVRTC compilation failure). This layer runs
        # once per forward pass (not in the block loop), so the perf impact is negligible.
        embedder_quant_config = None if in_channels < 128 else quant_config
        self.x_embedder = Linear(
            in_channels,
            self.inner_dim,
            bias=True,
            dtype=dtype,
            quant_config=embedder_quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )

        # Dual-stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    quant_config=quant_config,
                    skip_create_weights=skip_create_weights,
                    force_dynamic_quant=force_dynamic_quant,
                    config=model_config,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        # Single-stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    quant_config=quant_config,
                    skip_create_weights=skip_create_weights,
                    force_dynamic_quant=force_dynamic_quant,
                    config=model_config,
                    layer_idx=num_layers + i,  # Continue numbering from dual-stream blocks
                )
                for i in range(num_single_layers)
            ]
        )

        # Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )
        # TRT-LLM Linear for quantization support
        self.proj_out = Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.Tensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: Latent image tokens (batch, seq_len, in_channels)
            encoder_hidden_states: T5 text embeddings (batch, txt_seq_len, joint_attention_dim)
            pooled_projections: CLIP pooled text embeddings (batch, pooled_projection_dim)
            timestep: Timestep tensor (batch,)
            img_ids: Image position IDs (seq_len, 3) or (batch, seq_len, 3)
            txt_ids: Text position IDs (txt_seq_len, 3) or (batch, txt_seq_len, 3)
            guidance: Guidance scale tensor (batch,) for FLUX.1-dev
            joint_attention_kwargs: Additional kwargs for attention
            return_dict: Whether to return dict or tuple

        Returns:
            Noise prediction tensor of shape (batch, seq_len, patch_size^2 * out_channels)
        """
        # Embed inputs (contiguous needed for FP8 quantize ops)
        hidden_states = self.x_embedder(hidden_states.contiguous())

        # Scale timestep (FLUX convention: multiply by 1000)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        # Compute timestep + guidance + text embedding
        if self.config.guidance_embeds and guidance is not None:
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)

        # Embed text
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Handle 3D IDs (batch dimension) - deprecated format
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        # Compute RoPE embeddings
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # Dual-stream blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Single-stream blocks
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return {"sample": output}

    def load_weights(self, weights: dict) -> None:
        """Load weights into the transformer.

        Args:
            weights: Dictionary of parameter name -> tensor
        """
        from tqdm import tqdm

        from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader

        # Map fused QKV layer names to original HF checkpoint names
        # HF checkpoint has separate to_q, to_k, to_v / add_q_proj, add_k_proj, add_v_proj
        # We fuse them into to_qkv / add_qkv_proj for better performance
        params_map = {
            "to_qkv": ["to_q", "to_k", "to_v"],
            "add_qkv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
        }

        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)

        for name, module in tqdm(self.named_modules(), desc="Loading weights"):
            # Create weights for modules with skip_create_weights_in_init=True
            # This must be done before loading weights (following Wan pattern)
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)

                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
            else:
                module_weights = loader.filter_weights(name, weights)
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(
                            module_weights[param_name].to(self.model_config.torch_dtype)
                        )

    def post_load_weights(self) -> None:
        """Call post_load_weights on all Linear modules and convert embedders to target dtype."""
        # Convert time_text_embed components to target dtype
        target_dtype = self.model_config.torch_dtype
        if hasattr(self, "time_text_embed"):
            if hasattr(self.time_text_embed, "timestep_embedder"):
                self.time_text_embed.timestep_embedder.to(target_dtype)
            if hasattr(self.time_text_embed, "text_embedder"):
                self.time_text_embed.text_embedder.to(target_dtype)
            if hasattr(self.time_text_embed, "guidance_embedder"):
                self.time_text_embed.guidance_embedder.to(target_dtype)

        # Call post_load_weights on all Linear modules
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
