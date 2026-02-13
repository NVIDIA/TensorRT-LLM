"""FLUX.2 attention utilities: position embeddings, rotary encoding, and parallel self-attention.

Key Components:
- Flux2ParallelSelfAttention: Fused QKV+MLP for single-stream blocks (48 blocks)
- Flux2PosEmbed: 4-axis rotary position embeddings
- Flux2SwiGLU: SwiGLU activation for FLUX.2 FFN

Key Differences from FLUX.1:
- 48 heads (vs 24 in FLUX.1)
- 6144 inner_dim (vs 3072 in FLUX.1)
- 4-axis RoPE [32,32,32,32] (vs 3-axis [16,56,56] in FLUX.1)
- theta=2000 (vs theta=10000 in FLUX.1)
- Fused QKV+MLP projection in single-stream blocks
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention_backend.utils import create_attention
from tensorrt_llm._torch.visual_gen.modules.attention import _per_head_norm

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


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
    - QKV: 3 x 6144 = 18432
    - MLP: 2 x 18432 = 36864 (for SwiGLU)
    - Fused projection: 6144 -> 55296 (18432 + 36864)
    - Output: concat(6144, 18432) -> 6144
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

        # Per-head QK normalization using TRT-LLM RMSNorm
        self.norm_q = RMSNorm(hidden_size=dim_head, eps=eps, has_weights=True)
        self.norm_k = RMSNorm(hidden_size=dim_head, eps=eps, has_weights=True)

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

        # Per-head QK normalization (reshape to 2D for TRT-LLM RMSNorm)
        query = _per_head_norm(query, self.norm_q)
        key = _per_head_norm(key, self.norm_k)

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
