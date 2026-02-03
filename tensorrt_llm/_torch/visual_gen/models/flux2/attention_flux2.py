"""FLUX.2 attention modules with joint attention mechanism.

FLUX.2 uses a unique joint attention mechanism where:
1. Image and text tokens get separate Q/K/V projections
2. QK normalization is applied per-head using RMSNorm
3. Q and K are concatenated (text + image) before attention
4. RoPE is applied to the concatenated Q and K (4-axis: [32,32,32,32])
5. After attention, outputs are split back into image and text

Key Components:
- Flux2Attention: Joint attention for dual-stream blocks (8 blocks)
- Flux2ParallelSelfAttention: Fused QKV+MLP for single-stream blocks (48 blocks)
- Flux2PosEmbed: 4-axis rotary position embeddings

Key Differences from FLUX.1:
- 48 heads (vs 24 in FLUX.1)
- 6144 inner_dim (vs 3072 in FLUX.1)
- 4-axis RoPE [32,32,32,32] (vs 3-axis [16,56,56] in FLUX.1)
- theta=2000 (vs theta=10000 in FLUX.1)
- Fused QKV+MLP projection in single-stream blocks
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

# =============================================================================
# Custom RMSNorm (avoids PyTorch version compatibility issues with fill_())
# =============================================================================


class SimpleRMSNorm(nn.Module):
    """Simple RMSNorm implementation compatible with all PyTorch versions.

    PyTorch's nn.RMSNorm has a bug in some versions where reset_parameters()
    calls fill_() with 3 arguments instead of 2. This simple implementation
    avoids that issue.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight.float()
        return x.to(dtype)


# =============================================================================
# Activation Functions
# =============================================================================


class Flux2SwiGLU(nn.Module):
    """SwiGLU activation function used in FLUX.2 FFN.

    FLUX.2 uses gate_fn(x1) * x2 (different from standard SwiGLU which is x * gate_fn(gate)).
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


# =============================================================================
# Position Embedding
# =============================================================================


class Flux2PosEmbed(nn.Module):
    """4-axis RoPE position embedding for FLUX.2.

    FLUX.2 uses 4 axes: [32, 32, 32, 32] = 128 head_dim
    FLUX.1 uses 3 axes: [16, 56, 56] = 128 head_dim
    """

    def __init__(self, theta: float = 2000.0, axes_dim: Tuple[int, ...] = (32, 32, 32, 32)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate RoPE embeddings from position IDs.

        Args:
            ids: Position IDs of shape [seq_len, num_axes]

        Returns:
            Tuple of (freqs_cos, freqs_sin), each of shape [seq_len, head_dim]
        """
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()

        freqs_dtype = torch.float32 if ids.device.type in ("mps", "npu") else torch.float64

        for i in range(n_axes):
            cos, sin = self._get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i], self.theta, freqs_dtype
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)

        return freqs_cos, freqs_sin

    def _get_1d_rotary_pos_embed(
        self, dim: int, pos: torch.Tensor, theta: float, freqs_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D rotary position embeddings."""
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
        )
        freqs = torch.outer(pos.to(freqs_dtype), freqs)
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(pos.dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(pos.dtype)
        return cos, sin


def apply_rotary_emb(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to input tensor.

    Args:
        x: Input tensor of shape [batch, seq, heads, head_dim]
        freqs_cos: Cosine frequencies [seq, head_dim]
        freqs_sin: Sine frequencies [seq, head_dim]

    Returns:
        Tensor with rotary embedding applied
    """
    # x shape: [batch, seq, heads, head_dim]
    x_r = x.float()

    # Reshape for rotation: treat pairs of dims
    x1 = x_r[..., 0::2]
    x2 = x_r[..., 1::2]

    # Expand freqs: [seq, head_dim] -> [1, seq, 1, head_dim]
    cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    cos1 = cos[..., 0::2]
    sin1 = sin[..., 0::2]

    # Apply rotation
    out1 = x1 * cos1 - x2 * sin1
    out2 = x1 * sin1 + x2 * cos1

    # Interleave back
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out.to(x.dtype)


# =============================================================================
# Attention (for dual-stream blocks)
# =============================================================================


class Flux2Attention(nn.Module):
    """FLUX.2 joint attention for dual-stream blocks (matches HuggingFace).

    Processes image and text tokens jointly with QK RMSNorm.

    Architecture:
    - 48 attention heads
    - 128 head dimension
    - 6144 inner dimension
    - Fused QKV projections for image (to_qkv) and text (add_qkv_proj)
    - HF has separate Q/K/V weights, fused during weight loading via WeightMode.FUSED_QKV_LINEAR
    - QK normalization using RMSNorm
    - Joint attention over concatenated text + image tokens
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 48,
        dim_head: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: Optional[int] = None,
        elementwise_affine: bool = True,
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
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.use_bias = bias
        self.dropout = dropout
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

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
        self.norm_q = SimpleRMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = SimpleRMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        # Output projection (ModuleList to match HF)
        self.to_out = nn.ModuleList(
            [
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
            ]
        )

        # Text projections (for joint attention)
        if added_kv_proj_dim is not None:
            # TODO(AIGV): Migrate to TRT-LLM RMSNorm (same 4D tensor limitation)
            self.norm_added_q = SimpleRMSNorm(dim_head, eps=eps)
            self.norm_added_k = SimpleRMSNorm(dim_head, eps=eps)

            # Fused QKV projection for text tokens (dual-stream blocks only)
            # HF checkpoint has separate add_q_proj, add_k_proj, add_v_proj - fused during weight loading
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
            num_kv_heads=heads,  # No GQA in FLUX.2
            quant_config=quant_config,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: Image features [batch, img_seq, dim]
            encoder_hidden_states: Text features [batch, txt_seq, dim] (for joint attention)
            attention_mask: Optional attention mask
            image_rotary_emb: Tuple of (freqs_cos, freqs_sin)

        Returns:
            For self-attention: hidden_states [batch, img_seq, dim]
            For joint attention: (hidden_states, encoder_hidden_states)
        """
        # Fused QKV projection for image tokens (single matmul + chunk)
        qkv = self.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        # Reshape to [batch, seq, heads, head_dim]
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        # QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Text projections for joint attention (fused QKV)
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_qkv = self.add_qkv_proj(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

            encoder_query = encoder_query.unflatten(-1, (self.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # Concatenate for joint attention
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        # Get sequence length and batch size before reshape
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Check backend's preferred layout (following WAN pattern)
        backend_layout = getattr(self.attn, "preferred_layout", AttentionTensorLayout.NHD)

        if backend_layout == AttentionTensorLayout.HND:
            # Transpose: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        else:
            # Flatten for NHD layout: [batch, seq, heads, head_dim] -> [batch, seq, inner_dim]
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
            # Transpose back: [batch, heads, seq, head_dim] -> [batch, seq, inner_dim]
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        # NHD output is already [batch, seq, inner_dim]

        hidden_states = hidden_states.to(query.dtype)

        # Split if joint attention
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split(
                [encoder_seq_len, hidden_states.shape[1] - encoder_seq_len], dim=1
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        # Output projection
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


# =============================================================================
# Parallel Self-Attention (for single-stream blocks)
# =============================================================================


class Flux2ParallelSelfAttention(nn.Module):
    """FLUX.2 parallel self-attention for single-stream blocks (matches HuggingFace).

    Uses fused QKV+MLP projection: to_qkv_mlp_proj
    Output: concatenate attention + MLP outputs, then project with to_out

    This is a key architectural difference from FLUX.1:
    - FLUX.1: Separate attention and FFN
    - FLUX.2: Fused QKV+MLP projection for efficiency

    Dimensions:
    - input: 6144
    - QKV: 3 × 6144 = 18432
    - MLP: 2 × 18432 = 36864 (for SwiGLU)
    - Fused projection: 6144 → 55296 (18432 + 36864)
    - Output: concat(6144, 18432) → 6144
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 48,
        dim_head: int = 128,
        mlp_ratio: float = 3.0,
        bias: bool = False,
        eps: float = 1e-6,
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
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head

        # MLP dimensions
        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = 2  # For SwiGLU

        # Fused QKV + MLP projection
        qkv_dim = 3 * self.inner_dim
        mlp_in_dim = self.mlp_hidden_dim * self.mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(
            query_dim,
            qkv_dim + mlp_in_dim,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
            disable_deep_gemm=True,
        )

        # QK normalization (per-head RMSNorm)
        # TODO(AIGV): Migrate to TRT-LLM RMSNorm when diffusion Attention module is implemented
        # TRT-LLM RMSNorm doesn't support 4D tensors (batch, seq, heads, head_dim)
        # Solution: Implement apply_qk_norm() stage in diffusion Attention module per design doc
        self.norm_q = SimpleRMSNorm(dim_head, eps=eps)
        self.norm_k = SimpleRMSNorm(dim_head, eps=eps)

        # MLP activation (SwiGLU reduces dim by half: mlp_hidden_dim * 2 -> mlp_hidden_dim)
        self.mlp_act_fn = Flux2SwiGLU()

        # Combined output projection: [inner_dim + mlp_hidden_dim] -> [query_dim]
        # This matches HuggingFace which concatenates attn output + MLP output then projects
        self.to_out = Linear(
            self.inner_dim + self.mlp_hidden_dim,
            query_dim,
            bias=bias,
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
            num_kv_heads=heads,  # No GQA in FLUX.2
            quant_config=quant_config,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, dim]
            attention_mask: Optional attention mask
            image_rotary_emb: Tuple of (freqs_cos, freqs_sin)

        Returns:
            hidden_states [batch, seq, dim]
        """
        # Parallel QKV + MLP projection
        proj_out = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            proj_out, [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1
        )

        # Split QKV
        query, key, value = qkv.chunk(3, dim=-1)

        # Reshape to [batch, seq, heads, head_dim]
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        # QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        # Get sequence length and batch size before reshape
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Check backend's preferred layout (following WAN pattern)
        backend_layout = getattr(self.attn, "preferred_layout", AttentionTensorLayout.NHD)

        if backend_layout == AttentionTensorLayout.HND:
            # Transpose: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        else:
            # Flatten for NHD layout: [batch, seq, heads, head_dim] -> [batch, seq, inner_dim]
            query = query.flatten(2)
            key = key.flatten(2)
            value = value.flatten(2)

        # Attention via backend (TRTLLM or VANILLA)
        attn_output = self.attn.forward(
            q=query,
            k=key,
            v=value,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Reverse the layout transformation
        if backend_layout == AttentionTensorLayout.HND:
            # Transpose back: [batch, heads, seq, head_dim] -> [batch, seq, inner_dim]
            attn_output = attn_output.transpose(1, 2).flatten(2, 3)
        # NHD output is already [batch, seq, inner_dim]

        attn_output = attn_output.to(query.dtype)

        # MLP path: apply SwiGLU activation (reduces from mlp_hidden_dim*2 to mlp_hidden_dim)
        mlp_output = self.mlp_act_fn(mlp_hidden_states)

        # Concatenate attention and MLP outputs, then project
        combined = torch.cat([attn_output, mlp_output], dim=-1)
        hidden_states = self.to_out(combined)

        return hidden_states
