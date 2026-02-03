"""FLUX attention modules with joint attention mechanism.

FLUX uses a unique joint attention mechanism where:
1. Image and text tokens get separate Q/K/V projections
2. QK normalization is applied per-head using RMSNorm
3. Q and K are concatenated (text + image) before attention
4. RoPE is applied to the concatenated Q and K
5. After attention, outputs are split back into image and text

Key Components:
- FluxAttention: Joint attention with separate image/text projections
- FluxPosEmbed: 2D rotary position embeddings for (txt, h, w) axes
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


class FluxRMSNorm(nn.Module):
    """Simple RMSNorm without reset_parameters to avoid TRT-LLM dispatch issues.

    TRT-LLM's torch dispatch handler intercepts tensor operations, which causes
    issues with PyTorch's FluxRMSNorm.reset_parameters() that calls fill_().
    This class creates weights directly to avoid the issue.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            # Create weight directly without reset_parameters
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        # Cast to float32 for numerical stability, then back to input dtype
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x_fp32 * rms * self.weight
        return result.to(input_dtype)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
    freqs_dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 1D rotary position embeddings.

    Args:
        dim: Embedding dimension (must be even)
        pos: Position tensor of shape (seq_len,)
        theta: RoPE theta parameter
        use_real: Return (cos, sin) instead of complex exp
        repeat_interleave_real: Interleave or repeat cos/sin
        freqs_dtype: Dtype for frequency computation

    Returns:
        Tuple of (cos, sin) tensors each of shape (seq_len, dim)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # Compute frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))

    # Outer product: (seq_len, dim/2)
    freqs = torch.outer(pos.to(freqs_dtype), freqs)

    if use_real:
        if repeat_interleave_real:
            # Repeat each frequency: [f0, f0, f1, f1, ...]
            freqs = freqs.repeat_interleave(2, dim=-1)
        else:
            # Repeat pattern: [f0, f1, ..., f0, f1, ...]
            freqs = freqs.repeat(1, 2)

        cos = freqs.cos().to(pos.dtype)
        sin = freqs.sin().to(pos.dtype)
        return cos, sin
    else:
        # Return complex exponential
        return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, seq_len, heads, dim) or (batch, seq_len, dim)
        freqs_cis: Tuple of (cos, sin) from get_1d_rotary_pos_embed or FluxPosEmbed
        sequence_dim: Dimension containing sequence length (default: 1)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    cos, sin = freqs_cis

    # Cast to input dtype to avoid dtype mismatch
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    # Handle different input shapes
    ndim = x.ndim
    if ndim == 4:
        # (batch, seq, heads, dim)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif ndim == 3:
        # (batch, seq, dim)
        cos = cos.unsqueeze(0)  # (1, seq, dim)
        sin = sin.unsqueeze(0)

    # Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    x_rotated = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)

    return x * cos + x_rotated * sin


class FluxPosEmbed(nn.Module):
    """2D Rotary Position Embedding for FLUX.

    FLUX uses 3-axis RoPE encoding with dimensions:
    - txt_dim (16): For text sequence marker
    - h_dim (56): For height positions
    - w_dim (56): For width positions

    Total: 16 + 56 + 56 = 128 = attention_head_dim

    Position IDs format:
    - txt_ids: (seq_len, 3) with zeros (text has no spatial position)
    - img_ids: (seq_len, 3) with [0, h_pos, w_pos] for each patch

    The concatenated IDs (txt_ids + img_ids) are passed to forward().
    """

    def __init__(self, theta: int = 10000, axes_dim: List[int] = None):
        """Initialize FluxPosEmbed.

        Args:
            theta: Base for exponential frequency computation
            axes_dim: Dimensions for each axis [txt_dim, h_dim, w_dim]
                     Default: [16, 56, 56] which sums to 128
        """
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim if axes_dim is not None else [16, 56, 56]

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings from position IDs.

        Args:
            ids: Position IDs tensor of shape (seq_len, 3)
                 Column 0: text marker (0 for text, 0 for image)
                 Column 1: height position
                 Column 2: width position

        Returns:
            Tuple of (freqs_cos, freqs_sin), each of shape (seq_len, head_dim)
        """
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()

        # Determine frequency dtype based on device
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        # Compute RoPE for each axis
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        # Concatenate along dimension axis
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)

        return freqs_cos, freqs_sin


class FluxAttention(nn.Module):
    """Joint attention module for FLUX transformer.

    FLUX attention differs from standard attention:
    1. Separate projections for image (to_q, to_k, to_v) and text (add_q_proj, etc.)
    2. QK normalization using RMSNorm per head
    3. Concatenate text + image tokens before attention
    4. Apply RoPE to concatenated Q, K
    5. Split output back into image and text parts

    For dual-stream blocks: Returns (img_attn_output, txt_attn_output)
    For single-stream blocks: pre_only=True, no output projection, no text projections
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: Optional[int] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
    ):
        """Initialize FluxAttention.

        Args:
            query_dim: Input dimension for image tokens
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout probability
            bias: Use bias in Q/K/V projections
            added_kv_proj_dim: Dimension of text tokens (enables joint attention)
            added_proj_bias: Use bias in text projections
            out_bias: Use bias in output projection
            eps: Epsilon for RMSNorm
            out_dim: Output dimension (defaults to query_dim)
            pre_only: If True, skip output projections (for single-stream blocks)
            elementwise_affine: Use learnable parameters in RMSNorm
            dtype: Data type for linear layers (e.g., torch.bfloat16)
            quant_config: Quantization config for FP8/NVFP4
            skip_create_weights: Skip weight creation in init (for MetaInit)
            force_dynamic_quant: Force dynamic quantization
            config: DiffusionModelConfig for attention backend selection
            layer_idx: Layer index for attention backend
        """
        super().__init__()

        # Store config for attention backend
        self.config = config
        self.layer_idx = layer_idx

        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.dropout = dropout
        self.pre_only = pre_only
        self.added_kv_proj_dim = added_kv_proj_dim

        # Fused QKV projection for image tokens (single matmul instead of 3)
        # HF checkpoint has separate to_q, to_k, to_v - fused during weight loading via params_map
        self.to_qkv = Linear(
            query_dim,
            3 * self.inner_dim,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
            fused_weight_shard_indices_mapping={
                "q": (0, self.inner_dim),
                "k": (self.inner_dim, self.inner_dim),
                "v": (2 * self.inner_dim, self.inner_dim),
            },
        )

        # QK normalization (per-head RMSNorm)
        # TODO(AIGV): Migrate to TRT-LLM RMSNorm when diffusion Attention module is implemented
        # TRT-LLM RMSNorm doesn't support 4D tensors (batch, seq, heads, head_dim)
        # Solution: Implement apply_qk_norm() stage in diffusion Attention module per design doc
        self.norm_q = FluxRMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = FluxRMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        # Output projection (unless pre_only) - TRT-LLM Linear
        if not self.pre_only:
            self.to_out = nn.Sequential(
                Linear(
                    self.inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    dtype=dtype,
                    quant_config=quant_config,
                    skip_create_weights_in_init=skip_create_weights,
                    force_dynamic_quantization=force_dynamic_quant,
                    disable_deep_gemm=True,
                ),
                nn.Dropout(dropout),
            )

        # Fused QKV projection for text tokens (dual-stream blocks only)
        # HF checkpoint has separate add_q_proj, add_k_proj, add_v_proj - fused during weight loading
        if added_kv_proj_dim is not None:
            self.add_qkv_proj = Linear(
                added_kv_proj_dim,
                3 * self.inner_dim,
                bias=added_proj_bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights,
                force_dynamic_quantization=force_dynamic_quant,
                disable_deep_gemm=True,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR
                ),
                fused_weight_shard_indices_mapping={
                    "q": (0, self.inner_dim),
                    "k": (self.inner_dim, self.inner_dim),
                    "v": (2 * self.inner_dim, self.inner_dim),
                },
            )

            # QK normalization for text
            # TODO(AIGV): Migrate to TRT-LLM RMSNorm (same 4D tensor limitation as above)
            self.norm_added_q = FluxRMSNorm(dim_head, eps=eps)
            self.norm_added_k = FluxRMSNorm(dim_head, eps=eps)

            # Text output projection - TRT-LLM Linear
            self.to_add_out = Linear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights,
                force_dynamic_quantization=force_dynamic_quant,
                disable_deep_gemm=True,
            )

        # Create attention backend (TRTLLM or VANILLA)
        attn_backend = "VANILLA"  # Default
        if config is not None and hasattr(config, "attention"):
            attn_backend = config.attention.backend

        self.attn = create_attention(
            backend=attn_backend,
            layer_idx=layer_idx,
            num_heads=heads,
            head_dim=dim_head,
            num_kv_heads=heads,  # No GQA in FLUX
            quant_config=quant_config,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass of joint attention.

        Args:
            hidden_states: Image tokens (batch, img_seq_len, dim)
            encoder_hidden_states: Text tokens (batch, txt_seq_len, dim) for dual-stream
            attention_mask: Optional attention mask
            image_rotary_emb: Tuple of (cos, sin) for RoPE

        Returns:
            For dual-stream (encoder_hidden_states provided):
                Tuple of (img_attn_output, txt_attn_output)
            For single-stream (no encoder_hidden_states):
                Attention output tensor
        """
        batch_size = hidden_states.shape[0]

        # Fused QKV projection for image tokens (single matmul + chunk)
        qkv = self.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        # Reshape: (batch, seq, inner_dim) -> (batch, seq, heads, head_dim)
        query = query.view(batch_size, -1, self.heads, self.head_dim)
        key = key.view(batch_size, -1, self.heads, self.head_dim)
        value = value.view(batch_size, -1, self.heads, self.head_dim)

        # Apply QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Process text tokens if provided (dual-stream joint attention)
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            txt_seq_len = encoder_hidden_states.shape[1]

            # Fused QKV projection for text tokens (single matmul + chunk)
            encoder_qkv = self.add_qkv_proj(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

            # Reshape
            encoder_query = encoder_query.view(batch_size, -1, self.heads, self.head_dim)
            encoder_key = encoder_key.view(batch_size, -1, self.heads, self.head_dim)
            encoder_value = encoder_value.view(batch_size, -1, self.heads, self.head_dim)

            # Apply QK normalization for text
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # Concatenate text + image: (batch, txt_seq + img_seq, heads, head_dim)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply RoPE to concatenated Q, K
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Get sequence length before reshape
        seq_len = query.shape[1]

        # Check backend's preferred layout (following WAN pattern)
        backend_layout = getattr(self.attn, "preferred_layout", AttentionTensorLayout.NHD)

        if backend_layout == AttentionTensorLayout.HND:
            # Transpose for attention: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        else:
            # Flatten for NHD layout: (batch, seq, heads, head_dim) -> (batch, seq, inner_dim)
            query = query.flatten(2)
            key = key.flatten(2)
            value = value.flatten(2)

        # Attention via backend (TRTLLM or VANILLA)
        hidden_states = self.attn.forward(
            q=query,
            k=key,
            v=value,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Reverse the layout transformation
        if backend_layout == AttentionTensorLayout.HND:
            # Transpose back: (batch, heads, seq, head_dim) -> (batch, seq, inner_dim)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            hidden_states = hidden_states.view(batch_size, -1, self.inner_dim)
        # NHD output is already (batch, seq, inner_dim)

        hidden_states = hidden_states.to(query.dtype)

        # Split and project outputs
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            # Split back into text and image parts
            encoder_hidden_states_out, hidden_states = hidden_states.split(
                [txt_seq_len, hidden_states.shape[1] - txt_seq_len], dim=1
            )

            # Apply output projections
            if not self.pre_only:
                hidden_states = self.to_out(hidden_states)
            encoder_hidden_states_out = self.to_add_out(encoder_hidden_states_out)

            return hidden_states, encoder_hidden_states_out
        else:
            # Single-stream: just return attention output
            if not self.pre_only:
                hidden_states = self.to_out(hidden_states)
            return hidden_states


def prepare_flux_image_ids(
    height: int,
    width: int,
    patch_size: int = 2,
    vae_scale_factor: int = 8,
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare position IDs for image latents in FLUX.

    FLUX packs 2x2 patches, so the effective grid is (height/16, width/16).

    Args:
        height: Image height in pixels
        width: Image width in pixels
        patch_size: Packing patch size (default: 2)
        vae_scale_factor: VAE spatial downsampling factor (default: 8)
        device: Target device

    Returns:
        Position IDs tensor of shape (num_patches, 3) with columns [0, h_pos, w_pos]
    """
    # Compute latent dimensions after VAE and packing
    latent_h = height // (vae_scale_factor * patch_size)
    latent_w = width // (vae_scale_factor * patch_size)

    # Create position grid
    h_pos = torch.arange(latent_h, device=device)
    w_pos = torch.arange(latent_w, device=device)

    # Create meshgrid
    h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing="ij")

    # Flatten and stack: (num_patches, 3)
    # Column 0: text marker (0 for images)
    # Column 1: height position
    # Column 2: width position
    img_ids = torch.zeros(latent_h * latent_w, 3, device=device)
    img_ids[:, 1] = h_grid.flatten()
    img_ids[:, 2] = w_grid.flatten()

    return img_ids


def prepare_flux_text_ids(
    seq_len: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare position IDs for text tokens in FLUX.

    Text tokens have no spatial position, so all IDs are zeros.

    Args:
        seq_len: Text sequence length
        device: Target device

    Returns:
        Position IDs tensor of shape (seq_len, 3) with all zeros
    """
    return torch.zeros(seq_len, 3, device=device)
