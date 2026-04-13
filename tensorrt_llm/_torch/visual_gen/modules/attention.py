from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from ...modules.linear import Linear, WeightMode, WeightsLoadingConfig
from ...modules.rms_norm import RMSNorm
from ..attention_backend.interface import AttentionTensorLayout
from ..attention_backend.utils import create_attention

if TYPE_CHECKING:
    from ..config import DiffusionModelConfig


class QKVMode(str, Enum):
    FUSE_QKV = "fuse_qkv"
    FUSE_KV = "fuse_kv"
    SEPARATE_QKV = "separate"


# TODO: torch compile
def apply_rotary_emb(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    freqs_cos = freqs_cos.to(x.dtype)
    freqs_sin = freqs_sin.to(x.dtype)
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)  # [B, S, H, D/2]

    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]

    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    """Attention module for visual generation models."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qkv_mode: QKVMode = QKVMode.FUSE_QKV,
        qk_norm: bool = True,
        qk_norm_mode: str = "full",
        eps: float = 1e-6,
        bias: bool = True,
        interleave: bool = True,
        fuse_qk_norm_rope: Optional[bool] = None,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()

        config = config or DiffusionModelConfig()
        self.dtype = config.torch_dtype
        self.quant_config = config.quant_config
        self.skip_create_weights_in_init = config.skip_create_weights_in_init
        self.force_dynamic_quantization = config.force_dynamic_quantization
        self.mapping = getattr(config, "mapping", None)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.qkv_mode = QKVMode(qkv_mode) if isinstance(qkv_mode, str) else qkv_mode
        self.bias = bias

        # Fused QK Norm + RoPE: each model class opts in via fuse_qk_norm_rope.
        # Default: enable for per_head norm only (FLUX). Full-dim not yet supported.
        if fuse_qk_norm_rope is not None:
            self.fuse_qk_norm_rope = fuse_qk_norm_rope
        else:
            self.fuse_qk_norm_rope = qk_norm_mode != "full"
        self.interleave = interleave

        # Select compute backend (orthogonal to parallelism)
        vgm = config.visual_gen_mapping
        ulysses_size = vgm.ulysses_size if vgm else 1
        base_backend = config.attention.backend

        # TRTLLM doesn't support cross-attention (different Q/KV seq lengths); fall back to VANILLA
        if self.qkv_mode == QKVMode.SEPARATE_QKV and base_backend == "TRTLLM":
            backend_name = "VANILLA"
        else:
            backend_name = base_backend
        self.attn_backend = backend_name
        self.qk_norm = qk_norm
        self.qk_norm_mode = qk_norm_mode
        self.layer_idx = layer_idx if layer_idx is not None else 0
        self.eps = eps

        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim

        self._init_qkv_proj()

        if self.qk_norm:
            # "full": norm over all heads combined (e.g. WAN, dim=q_dim)
            # "per_head": norm over each head independently (e.g. FLUX, dim=head_dim)
            q_norm_dim = self.head_dim if qk_norm_mode == "per_head" else self.q_dim
            k_norm_dim = self.head_dim if qk_norm_mode == "per_head" else self.kv_dim
            self.norm_q = RMSNorm(
                hidden_size=q_norm_dim, eps=self.eps, dtype=self.dtype, has_weights=True
            )
            self.norm_k = RMSNorm(
                hidden_size=k_norm_dim, eps=self.eps, dtype=self.dtype, has_weights=True
            )

        # TODO: Use weight mapper to create just a Linear module
        self.to_out = nn.ModuleList(
            [
                Linear(
                    self.q_dim,
                    self.hidden_size,
                    bias=self.bias,
                    dtype=self.dtype,
                    mapping=self.mapping,
                    quant_config=self.quant_config,
                    skip_create_weights_in_init=self.skip_create_weights_in_init,
                    force_dynamic_quantization=self.force_dynamic_quantization,
                )
            ]
        )

        # Compute head counts for the backend
        # Ulysses shards heads across workers; inner backend sees sharded count
        if ulysses_size > 1 and self.qkv_mode != QKVMode.SEPARATE_QKV:
            backend_num_heads = self.num_attention_heads // ulysses_size
            backend_num_kv_heads = self.num_key_value_heads // ulysses_size
        else:
            backend_num_heads = self.num_attention_heads
            backend_num_kv_heads = self.num_key_value_heads

        # Create compute backend
        self.attn = create_attention(
            backend=backend_name,
            layer_idx=self.layer_idx,
            num_heads=backend_num_heads,
            head_dim=self.head_dim,
            num_kv_heads=backend_num_kv_heads,
            quant_config=self.quant_config,
            dtype=self.dtype,
        )

        # Wrap with parallelism strategies (orthogonal to backend choice)
        if ulysses_size > 1 and self.qkv_mode != QKVMode.SEPARATE_QKV:
            from ..attention_backend.parallel import UlyssesAttention

            self.attn = UlyssesAttention(
                inner_backend=self.attn,
                process_group=vgm.ulysses_group,
            )

    def _init_qkv_proj(self) -> None:
        if self.qkv_mode == QKVMode.FUSE_QKV:
            qkv_out_dim = self.q_dim + 2 * self.kv_dim
            self.qkv_proj = Linear(
                self.hidden_size,
                qkv_out_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR
                ),
                fused_weight_shard_indices_mapping={
                    "q": (0, self.q_dim),
                    "k": (self.q_dim, self.kv_dim),
                    "v": (self.q_dim + self.kv_dim, self.kv_dim),
                },
            )
        else:
            self.to_q = Linear(
                self.hidden_size,
                self.q_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )
            self.to_k = Linear(
                self.hidden_size,
                self.kv_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )
            self.to_v = Linear(
                self.hidden_size,
                self.kv_dim,
                bias=self.bias,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )

    def get_qkv(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.qkv_mode == QKVMode.FUSE_QKV:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            kv_source = (
                encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            )
            q = self.to_q(hidden_states)
            k = self.to_k(kv_source)
            v = self.to_v(kv_source)
        return q, k, v

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        return q, k

    def apply_qk_norm_rope(
        self,
        qkv: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        num_txt_tokens: int = -1,
        q_add_weight: Optional[torch.Tensor] = None,
        k_add_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply fused QK Norm + RoPE in-place on packed QKV tensor."""
        cos_2d = freqs_cos.reshape(-1, self.head_dim).float().contiguous()
        sin_2d = freqs_sin.reshape(-1, self.head_dim).float().contiguous()

        B, S, D = qkv.shape
        assert cos_2d.shape == (S, self.head_dim), (
            f"cos_emb shape mismatch: expected [{S}, {self.head_dim}], got {list(cos_2d.shape)}"
        )
        qkv_2d = qkv.view(B * S, D)
        cos_tiled = cos_2d.repeat(B, 1) if B > 1 else cos_2d
        sin_tiled = sin_2d.repeat(B, 1) if B > 1 else sin_2d

        # Dual-stream batch correction: when B>1 and dual-stream is active,
        # the kernel uses modulo (tokenIdx % tokens_per_batch) to find the
        # local position within each batch element for the text/image boundary.
        # 0 = no dual-stream (single-stream or batch=1).
        tokens_per_batch = S if num_txt_tokens > 0 else 0

        torch.ops.trtllm.fused_dit_qk_norm_rope(
            qkv_2d,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.eps,
            self.norm_q.weight,
            self.norm_k.weight,
            q_add_weight,
            k_add_weight,
            cos_tiled,
            sin_tiled,
            num_txt_tokens,
            self.interleave,
            tokens_per_batch,
        )

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Call attention backend with appropriate tensor layout.

        Dimensions are derived from tensor shapes. Extra ``**kwargs``
        (e.g. ``attention_mask``) are forwarded to the backend.

        Two layout paths:
        1. HND backends (VANILLA): [B, S, H*D] -> [B, H, S, D]
        2. NHD backends (TRTLLM, UlyssesAttention): [B, S, H*D] -> [B, S, H, D]
        """
        backend_layout = getattr(self.attn, "preferred_layout", AttentionTensorLayout.NHD)

        batch_size = q.shape[0]

        # Reshape inputs: [B, S, H*D] -> backend's preferred 4D layout
        if backend_layout == AttentionTensorLayout.HND:
            q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim)
            k = k.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, -1, self.num_key_value_heads, self.head_dim)

        out = self.attn.forward(q=q, k=k, v=v, **kwargs)

        # Flatten back to [B, S, H*D]
        if backend_layout == AttentionTensorLayout.HND:
            return out.transpose(1, 2).flatten(2)
        else:
            return out.flatten(2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert hidden_states.ndim == 3, "hidden_states must be a 3D tensor"
        batch_size, seq_len = hidden_states.shape[:2]
        kv_seq_len = (
            encoder_hidden_states.shape[1] if encoder_hidden_states is not None else seq_len
        )

        # Fused path: QKV projection → fused QK norm + RoPE → attention
        if (
            self.fuse_qk_norm_rope
            and freqs is not None
            and self.qkv_mode == QKVMode.FUSE_QKV
            and self.qk_norm
        ):
            qkv = self.qkv_proj(hidden_states)
            freqs_cos, freqs_sin = freqs
            self.apply_qk_norm_rope(qkv, freqs_cos, freqs_sin)
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
            out = self._attn_impl(q, k, v)
            return self.to_out[0](out)

        # Unfused path: separate QK norm → separate RoPE → attention
        q, k, v = self.get_qkv(hidden_states, encoder_hidden_states)
        q, k = self.apply_qk_norm(q, k)

        # Apply RoPE if provided (model handles RoPE, not attention backend)
        if freqs is not None:
            freqs_cos, freqs_sin = freqs
            q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)  # [B, S, H, D]
            k = k.view(batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim)
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
            q = q.flatten(2)
            k = k.flatten(2)

        out = self._attn_impl(q, k, v)
        out = self.to_out[0](out)
        return out
