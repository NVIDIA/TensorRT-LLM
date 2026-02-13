"""FLUX.2 Transformer model implementation (Native TRT-LLM).

FLUX.2 has a DIFFERENT architecture from FLUX.1:
- Different modulation: Flux2Modulation with mod_param_sets
- Different embedding: time_guidance_embed (timestep + guidance combined)
- Different FFN: Flux2FeedForward with Flux2SwiGLU
- Different single-stream: Fused QKV+MLP projection (to_qkv_mlp_proj)

This module provides a native implementation matching HuggingFace diffusers exactly.
All linear layers use bias=False to match HF weights.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.models.flux.attention_flux2 import (
    Flux2ParallelSelfAttention,
    Flux2PosEmbed,
    Flux2SwiGLU,
)
from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import (
    AdaLayerNormContinuous,
    _remap_checkpoint_keys,
)
from tensorrt_llm._torch.visual_gen.modules.attention import FluxJointAttention

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

# =============================================================================
# Time + Guidance Embedding (matches HuggingFace structure exactly)
# =============================================================================


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    """Combined timestep and guidance embedding for FLUX.2 (matches HuggingFace exactly).

    Structure:
    - time_proj: Sinusoidal projection (Timesteps)
    - timestep_embedder: 2-layer MLP (TimestepEmbedding)
    - guidance_embedder: 2-layer MLP (TimestepEmbedding)

    Output = timestep_emb + guidance_emb
    """

    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Sinusoidal projection for timesteps (shared for both timestep and guidance)
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        # Separate embedders for timestep and guidance
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )

        self.guidance_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: [batch] timestep values (already scaled by 1000)
            guidance: [batch] guidance scale values (already scaled by 1000)

        Returns:
            Combined embedding [batch, embedding_dim]
        """
        # Sinusoidal projection
        timesteps_proj = self.time_proj(timestep)
        guidance_proj = self.time_proj(guidance)

        # MLP embedding
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))
        guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))

        # Combine by addition (not concatenation)
        return timesteps_emb + guidance_emb


# =============================================================================
# Modulation
# =============================================================================


class Flux2Modulation(nn.Module):
    """FLUX.2 modulation layer (matches HuggingFace exactly).

    Projects temb to shift/scale/gate for layer normalization and attention gating.
    """

    def __init__(
        self,
        dim: int,
        mod_param_sets: int = 2,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = Linear(
            dim,
            dim * 3 * mod_param_sets,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        self.act_fn = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """
        Args:
            temb: Time embedding [batch, dim]

        Returns:
            Tuple of mod_param_sets 3-tuples, each containing (shift, scale, gate)
            Each tensor has shape [batch, 1, dim]
        """
        mod = self.act_fn(temb)
        mod = self.linear(mod)

        if mod.ndim == 2:
            mod = mod.unsqueeze(1)

        # Split into 3*mod_param_sets chunks
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)

        # Return tuple of 3-tuples (shift, scale, gate)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


# =============================================================================
# Feed Forward
# =============================================================================


class Flux2FeedForward(nn.Module):
    """FLUX.2 feed-forward network with SwiGLU activation (matches HuggingFace).

    Structure: linear_in -> SwiGLU -> linear_out
    Note: linear_in outputs 2x inner_dim for the SwiGLU gate.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 3.0,
        inner_dim: Optional[int] = None,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # linear_in outputs 2x for SwiGLU (value + gate)
        self.linear_in = Linear(
            dim,
            inner_dim * 2,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        self.act_fn = Flux2SwiGLU()
        self.linear_out = Linear(
            inner_dim,
            dim_out,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


# =============================================================================
# Transformer Blocks
# =============================================================================


class Flux2TransformerBlock(nn.Module):
    """FLUX.2 dual-stream transformer block (matches HuggingFace).

    Processes image and text tokens with shared attention but separate FFN.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Layer norms (TRT-LLM - without elementwise affine, modulation provides scale/shift)
        self.norm1 = LayerNorm(hidden_size=dim, eps=eps, has_weights=False, has_bias=False)
        self.norm1_context = LayerNorm(hidden_size=dim, eps=eps, has_weights=False, has_bias=False)

        # Joint attention
        self.attn = FluxJointAttention(
            hidden_size=dim,
            num_attention_heads=num_attention_heads,
            head_dim=attention_head_dim,
            bias=bias,
            added_kv_proj_dim=dim,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            config=config,
            layer_idx=layer_idx,
        )

        # FFN for image stream
        self.ff = Flux2FeedForward(
            dim=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )
        # FFN for text stream
        self.ff_context = Flux2FeedForward(
            dim=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        img_mod: Tuple[Tuple[torch.Tensor, ...], ...],
        txt_mod: Tuple[Tuple[torch.Tensor, ...], ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Image features [batch, img_seq, dim]
            encoder_hidden_states: Text features [batch, txt_seq, dim]
            image_rotary_emb: Tuple of (freqs_cos, freqs_sin)
            img_mod: Image modulation ((shift1, scale1, gate1), (shift2, scale2, gate2))
            txt_mod: Text modulation ((shift1, scale1, gate1), (shift2, scale2, gate2))

        Returns:
            Tuple of (encoder_hidden_states, hidden_states)
        """
        # Unpack modulation parameters
        (img_shift1, img_scale1, img_gate1), (img_shift2, img_scale2, img_gate2) = img_mod
        (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = txt_mod

        # Save residuals
        img_residual = hidden_states
        txt_residual = encoder_hidden_states

        # Pre-norm + modulation
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states * (1 + img_scale1) + img_shift1

        encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states * (1 + txt_scale1) + txt_shift1

        # Joint attention
        attn_output, encoder_attn_output = self.attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Attention residual with gate
        hidden_states = img_residual + attn_output * img_gate1
        encoder_hidden_states = txt_residual + encoder_attn_output * txt_gate1

        # FFN
        img_residual = hidden_states
        txt_residual = encoder_hidden_states

        # Modulation for FFN (use scale2/shift2)
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states * (1 + img_scale2) + img_shift2

        encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states * (1 + txt_scale2) + txt_shift2

        # FFN with gate
        hidden_states = img_residual + self.ff(hidden_states) * img_gate2
        encoder_hidden_states = txt_residual + self.ff_context(encoder_hidden_states) * txt_gate2

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    """FLUX.2 single-stream transformer block (matches HuggingFace).

    Uses parallel attention and MLP with fused projection.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dim = dim

        # Layer norm (TRT-LLM - without elementwise affine)
        self.norm = LayerNorm(hidden_size=dim, eps=eps, has_weights=False, has_bias=False)

        # Parallel attention with fused QKV+MLP
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            mlp_ratio=mlp_ratio,
            bias=bias,
            eps=eps,
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
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        mod: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, dim]
            image_rotary_emb: Tuple of (freqs_cos, freqs_sin)
            mod: Modulation (shift, scale, gate)

        Returns:
            hidden_states [batch, seq, dim]
        """
        mod_shift, mod_scale, mod_gate = mod

        # Save residual
        residual = hidden_states

        # Pre-norm + modulation
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + mod_scale) + mod_shift

        # Parallel attention + MLP
        hidden_states = self.attn(hidden_states, image_rotary_emb=image_rotary_emb)

        # Residual with gate
        hidden_states = residual + hidden_states * mod_gate

        return hidden_states


# =============================================================================
# Main Model
# =============================================================================


class Flux2Transformer2DModel(nn.Module):
    """FLUX.2 Transformer model for image generation (Native TRT-LLM).

    This implements the full FLUX.2 architecture matching HuggingFace diffusers:
    - 8 dual-stream transformer blocks with joint attention
    - 48 single-stream transformer blocks with fused QKV+MLP
    - 4-axis RoPE position embeddings
    - Shared modulation layers for all blocks of same type
    """

    _supports_gradient_checkpointing = True

    def __init__(self, model_config: "DiffusionModelConfig"):
        """Initialize FLUX.2 transformer.

        Args:
            model_config: DiffusionModelConfig instance (from DiffusionModelLoader)
        """
        super().__init__()
        self.model_config = model_config

        # Extract pretrained config from model_config (following WAN/FLUX.1 pattern)
        pretrained_config = model_config.pretrained_config

        # Extract dtype and quantization config for Linear modules
        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        # Extract FLUX.2-specific parameters from pretrained config
        patch_size = getattr(pretrained_config, "patch_size", 1)
        in_channels = getattr(pretrained_config, "in_channels", 128)
        out_channels = getattr(pretrained_config, "out_channels", None)
        if out_channels is None:
            out_channels = in_channels  # Default to in_channels (like Flux2Config.__post_init__)
        num_layers = getattr(pretrained_config, "num_layers", 8)
        num_single_layers = getattr(pretrained_config, "num_single_layers", 48)
        attention_head_dim = getattr(pretrained_config, "attention_head_dim", 128)
        num_attention_heads = getattr(pretrained_config, "num_attention_heads", 48)
        mlp_ratio = getattr(pretrained_config, "mlp_ratio", 3.0)
        joint_attention_dim = getattr(pretrained_config, "joint_attention_dim", 15360)
        pooled_projection_dim = getattr(pretrained_config, "pooled_projection_dim", 5120)
        timestep_guidance_channels = getattr(pretrained_config, "timestep_guidance_channels", 256)
        guidance_embeds = getattr(pretrained_config, "guidance_embeds", True)
        axes_dims_rope = tuple(getattr(pretrained_config, "axes_dims_rope", [32, 32, 32, 32]))
        theta_rope = getattr(pretrained_config, "rope_theta", 2000.0)
        eps = getattr(pretrained_config, "eps", 1e-6)

        # Compute inner dimension
        inner_dim = num_attention_heads * attention_head_dim

        # Store key attributes (like FLUX.1)
        self.inner_dim = inner_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.guidance_embeds = guidance_embeds

        # Store config for compatibility (like FLUX.1 pattern)
        self.config = type(
            "Config",
            (),
            {
                "patch_size": patch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_layers": num_layers,
                "num_single_layers": num_single_layers,
                "attention_head_dim": attention_head_dim,
                "num_attention_heads": num_attention_heads,
                "mlp_ratio": mlp_ratio,
                "joint_attention_dim": joint_attention_dim,
                "pooled_projection_dim": pooled_projection_dim,
                "timestep_guidance_channels": timestep_guidance_channels,
                "guidance_embeds": guidance_embeds,
                "axes_dims_rope": axes_dims_rope,
                "theta_rope": theta_rope,
                "eps": eps,
                "inner_dim": inner_dim,
            },
        )()

        # Position embedding (4-axis RoPE)
        self.pos_embed = Flux2PosEmbed(
            theta=theta_rope,
            axes_dim=axes_dims_rope,
        )

        # Time + guidance embedding (small layers, optional quantization)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=inner_dim,
            bias=False,
            dtype=dtype,
        )

        # Modulation layers (shared across all blocks of same type)
        # mod_param_sets=2 for double stream (attn + ff)
        self.double_stream_modulation_img = Flux2Modulation(
            inner_dim,
            mod_param_sets=2,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            inner_dim,
            mod_param_sets=2,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )
        # mod_param_sets=1 for single stream (parallel attn+ff)
        self.single_stream_modulation = Flux2Modulation(
            inner_dim,
            mod_param_sets=1,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )

        # Input embedders
        # NOTE: x_embedder quantization is disabled when in_channels < 128.
        # FLUX.2 has in_channels=128 (OK), but future variants with smaller latent
        # channels would hit the same fp8_block_scaling_gemm NVRTC failure as FLUX.1
        # (in_channels=64). This layer runs once per forward pass, so the perf
        # impact is negligible.
        embedder_quant_config = None if in_channels < 128 else quant_config
        self.x_embedder = Linear(
            in_channels,
            inner_dim,
            bias=False,
            dtype=dtype,
            quant_config=embedder_quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )
        self.context_embedder = Linear(
            joint_attention_dim,
            inner_dim,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )

        # Dual-stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
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
                Flux2SingleTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
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
            inner_dim,
            inner_dim,
            elementwise_affine=False,
            eps=eps,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights=skip_create_weights,
            force_dynamic_quant=force_dynamic_quant,
        )
        self.proj_out = Linear(
            inner_dim,
            patch_size**2 * out_channels,
            bias=False,
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
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: Latent image features [batch, img_seq, in_channels]
            encoder_hidden_states: Text features [batch, txt_seq, joint_attention_dim]
            timestep: Diffusion timestep [batch]
            img_ids: Image position IDs [img_seq, num_axes] or [batch, img_seq, num_axes]
            txt_ids: Text position IDs [txt_seq, num_axes] or [batch, txt_seq, num_axes]
            guidance: Guidance scale [batch]
            joint_attention_kwargs: Additional kwargs for attention (unused)
            return_dict: Whether to return a dict

        Returns:
            Predicted noise [batch, img_seq, patch_size^2 * out_channels]
        """
        txt_seq_len = encoder_hidden_states.shape[1]

        # Embed inputs (contiguous needed for FP8 quantize ops)
        hidden_states = self.x_embedder(hidden_states.contiguous())
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Scale timestep and guidance (FLUX convention)
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000

        # Time + guidance embedding
        temb = self.time_guidance_embed(timestep, guidance)

        # Handle batched IDs
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        # Compute RoPE embeddings (4-axis)
        ids = torch.cat([txt_ids, img_ids], dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # Compute modulation parameters (shared across all blocks)
        img_mod = self.double_stream_modulation_img(temb)  # ((s1,sc1,g1), (s2,sc2,g2))
        txt_mod = self.double_stream_modulation_txt(temb)  # ((s1,sc1,g1), (s2,sc2,g2))
        single_mod = self.single_stream_modulation(temb)  # ((shift, scale, gate),)

        # Dual-stream blocks
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                img_mod=img_mod,
                txt_mod=txt_mod,
            )

        # Concatenate for single-stream blocks
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Single-stream blocks
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                image_rotary_emb=image_rotary_emb,
                mod=single_mod[0],  # Single tuple of (shift, scale, gate)
            )

        # Extract image features (discard text)
        hidden_states = hidden_states[:, txt_seq_len:, :]

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if return_dict:
            return {"sample": output}
        return (output,)

    def load_weights(self, weights: dict) -> None:
        """Load weights into the transformer.

        Args:
            weights: Dictionary of parameter name -> tensor (from safetensors)
        """
        from tqdm import tqdm

        from tensorrt_llm._torch.modules.linear import Linear
        from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader

        # Remap HF checkpoint keys to our module attribute names
        weights = _remap_checkpoint_keys(weights)

        # Map fused QKV layer names to original HF checkpoint names
        # HF checkpoint has separate to_q, to_k, to_v / add_q_proj, add_k_proj, add_v_proj
        # We fuse them into qkv_proj / add_qkv_proj for better performance
        params_map = {
            "add_qkv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
            "qkv_proj": ["to_q", "to_k", "to_v"],
        }

        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)

        for name, module in tqdm(self.named_modules(), desc="Loading FLUX.2 weights"):
            # Create weights for modules with skip_create_weights_in_init=True
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
            else:
                # For non-Linear modules, load weights directly
                module_weights = loader.filter_weights(name, weights)
                target_dtype = self.model_config.torch_dtype
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(module_weights[param_name].to(target_dtype))

    def post_load_weights(self) -> None:
        """Call post_load_weights on all Linear modules and convert embedders to target dtype."""
        # Convert time_guidance_embed components to target dtype
        target_dtype = self.model_config.torch_dtype
        if hasattr(self, "time_guidance_embed"):
            if hasattr(self.time_guidance_embed, "timestep_embedder"):
                self.time_guidance_embed.timestep_embedder.to(target_dtype)
            if hasattr(self.time_guidance_embed, "guidance_embedder"):
                self.time_guidance_embed.guidance_embedder.to(target_dtype)

        # Call post_load_weights on all Linear modules
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
